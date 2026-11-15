"""ONE final resolver-only capability, not paid/build/account authorization.

Operator decisions are source-bound facts supplied by the parent, not runtime
'approved' booleans. A review/desk-supplied admission digest is an external trust
anchor, like the source-overlay digest. The permanent exclusive claim consumes
03 even on exception/restart; nothing here deletes/releases/retries it. Only an
active issuer can redeem one anonymous live channel for its direct resolver child.
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
# R20: a transport type/persisted nonce cannot authenticate an active issuer.
# Redemption below is served synchronously by this process's LIVE registry.
import socket
import select
import time
import hmac
import fcntl
import contextlib

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
    permit.secret=secrets.token_bytes(32);permit.channel=None;permit.redemption_started=False
    permit.controller_entered=False;permit.native_started=False
    permit.claim={'attempt_id':ATTEMPT,'controller_pid':permit.pid,'claim_id':permit.nonce,
                  'controller_birth':process_birth(permit.pid),
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
    require(process_birth(permit.pid)==permit.claim['controller_birth'],'issuer PID birth changed')
    require(not os.path.lexists(marker_path(permit,'retired')),'issuer retired')
    current=runpy.run_path(str(HERE.parents[1]/'source_generation.py'))['validate_selection'](permit.manifest,permit.source['overlay_sha256'])
    data,root,raw=checked_go(permit.admission_path,permit.admission_sha,current,permit.reviewed_head,starting=starting)
    require(timestamp(permit.claim['slot_end_utc'])<=timestamp(permit.claim['claimed_utc'])+
            datetime.timedelta(seconds=LIMITS['slot_seconds']),'claimed lease exceeds75min')
    return data


def retire_permit(permit):
    # Revoke memory/channel FIRST, even if journaling/locking subsequently fails.
    if type(permit) is not _Permit:return
    _ACTIVE.pop(permit.nonce,None);permit.secret=b''
    if permit.channel is not None:permit.channel.close()
    with claim_lock(permit.path):
        path=marker_path(permit,'retired')
        if path.exists():
            require(json.loads(protected_bytes(path))['claim_sha256']==digest(permit.raw),'foreign retirement marker')
        else:exclusive_record(path,{'claim_sha256':digest(permit.raw),'retired_utc':utc().isoformat()})


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


def process_birth(pid):
    # Reuse the reviewed, scheduler-state-independent kernel identity binding.
    from darwin_process_identity import DarwinIdentity
    record=DarwinIdentity().record(pid)
    require(record is not None and record['state']=='live','issuer/child birth unavailable')
    return {k:record[k] for k in ('pid','ppid','start','exe','sid','uid')}


def marker_path(permit, kind):
    return permit.path.parent/(ATTEMPT+'.'+kind+'.json')


def exclusive_record(path, value):
    raw=(json.dumps(value,sort_keys=True)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(fd,'wb',closefd=False) as out:
            out.write(raw);out.flush();os.fsync(fd)
        parent=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(parent)
        finally:os.close(parent)
    finally:os.close(fd)


@contextlib.contextmanager
def claim_lock(path):
    # Never wait behind a stuck/foreign holder. A collision is fail-closed and
    # does not remove either marker or give this final attempt another chance.
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:os.close(fd)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


def signed(secret, stage, body):
    return {'body':body,'mac':hmac.digest(secret,stage.encode()+canonical(body),'sha256').hex()}


def authenticated(secret,stage,value):
    require(type(value) is dict and set(value)=={'body','mac'} and type(value['mac']) is str,'malformed live reply')
    require(hmac.compare_digest(value['mac'],signed(secret,stage,value['body'])['mac']),'live issuer authentication failed')
    return value['body']


def wait_io(channel, write, deadline, guard):
    while True:
        guard()
        remaining=deadline-time.monotonic()
        require(remaining>0,'live exchange work deadline exhausted')
        ready=select.select([] if write else [channel],[channel] if write else [],[],min(0.5,remaining))
        if ready[1 if write else 0]:return


def send_frame(channel, value, deadline, guard):
    raw=canonical(value);require(len(raw)<=2048,'live frame oversized')
    pending=memoryview(len(raw).to_bytes(4,'big')+raw)
    while pending:
        wait_io(channel,True,deadline,guard)
        try:count=channel.send(pending)
        except BlockingIOError:continue
        require(count>0,'live issuer channel closed');pending=pending[count:]


def receive_frame(channel,deadline,guard,prefilled=False):
    def exact(size):
        parts=b''
        while len(parts)<size:
            if not prefilled:wait_io(channel,False,deadline,guard)
            try:chunk=channel.recv(size-len(parts))
            except BlockingIOError:
                require(not prefilled,'initial live capability not prefilled')
                continue
            require(bool(chunk),'live issuer channel closed');parts+=chunk
        return parts
    size=int.from_bytes(exact(4),'big');require(0<size<=2048,'partial/oversized live frame')
    return runpy.run_path(str(HERE.parents[1]/'source_generation.py'))['load'](exact(size))


def issue_resolver_handoff(permit, deadline=None):
    validate_controller(permit)
    require(permit.native_started,'resolver handoff before owned native start')
    require(not permit.handoff_issued,'resolver handoff already consumed')
    permit.handoff_issued=True
    # Secret is NEVER in claim/markers/logs. It survives only in live memory and
    # the inherited anonymous capability; persistent public metadata cannot mint it.
    permit.exchange_deadline=min(time.monotonic()+LIMITS['work_seconds'],
        deadline if deadline is not None else float('inf'),
        time.monotonic()+(timestamp(permit.claim['slot_end_utc'])-utc()).total_seconds())
    parent,child=socket.socketpair();parent.setblocking(False);child.setblocking(False)
    permit.channel=parent
    try:
        frame={'claim_id':permit.nonce,'secret':permit.secret.hex(),
            'source_sha256':permit.source['overlay_sha256'],'admission_sha256':permit.admission_sha,
            'source_commit':permit.reviewed_head,'deadline':permit.exchange_deadline}
        raw=canonical(frame);require(len(raw)<=2048,'initial capability oversized')
        packet=len(raw).to_bytes(4,'big')+raw
        # No child/reader exists yet: one NONBLOCKING send or abort, never wait
        # for buffer space before the process that could drain it is spawned.
        require(parent.send(packet)==len(packet),'initial capability not fully prefilled')
        return child.detach()
    except BaseException:
        child.close();retire_permit(permit);raise


def redeem_once(permit, child_birth, request_sha):
    with claim_lock(permit.path):
        validate_controller(permit)
        require(process_birth(permit.pid)==permit.claim['controller_birth'],'issuer birth changed')
        require(not os.path.lexists(marker_path(permit,'retired')),'issuer retired')
        require(protected_bytes(permit.path)==permit.raw,'persistent claim changed')
        record={'claim_sha256':digest(permit.raw),'child_birth':child_birth,
                'request_sha256':request_sha,'admission_sha256':permit.admission_sha,
                'source_sha256':permit.source['overlay_sha256']}
        exclusive_record(marker_path(permit,'resolver-consumed'),record)
        return record


def serve_resolver_handoff(permit, child_pid, guard=lambda:None):
    """Synchronous live exchange: no listener, service, thread or retry lane.

    Only the exact issued endpoint + real _ACTIVE permit can reply. Trust is
    approved source execution, NOT protection against arbitrary same-UID Python
    implementing a counterfeit server or root rewriting/executing everything.
    """
    def live():
        validate_controller(permit);guard()
    try:
        live();require(permit.handoff_issued and permit.channel is not None,'no live issued endpoint')
        require(not permit.redemption_started,'redemption already attempted');permit.redemption_started=True
        request=authenticated(permit.secret,'request',receive_frame(permit.channel,permit.exchange_deadline,live))
        require(type(request) is dict and set(request)=={'challenge','child_birth'} and
                type(request['challenge']) is str and re.fullmatch('[0-9a-f]{64}',request['challenge']),'bad live request')
        birth=process_birth(child_pid)
        require(request['child_birth']==birth and birth['ppid']==permit.pid,'wrong/stale child PID birth')
        record=redeem_once(permit,birth,digest(canonical(request)))
        grant={'request_sha256':record['request_sha256'],'claim_sha256':record['claim_sha256']}
        send_frame(permit.channel,signed(permit.secret,'grant',grant),permit.exchange_deadline,live)
        ack=authenticated(permit.secret,'ack',receive_frame(permit.channel,permit.exchange_deadline,live))
        require(ack==grant,'wrong redemption acknowledgement')
        with claim_lock(permit.path):
            live();require(not os.path.lexists(marker_path(permit,'retired')),'issuer retired before effects')
            require(json.loads(protected_bytes(marker_path(permit,'resolver-consumed')))==record,'redemption record changed')
            final_commit=signed(permit.secret,'commit',grant)
        # Release the record lock BEFORE waking the child; its immediate effect
        # fence must not collide with this parent's already-completed commit.
        send_frame(permit.channel,final_commit,permit.exchange_deadline,live)
        # Keep peer open: retirement closes it BEFORE fallible marker I/O. Child
        # effect guards detect EOF even if durable retirement recording fails.
        return record
    except BaseException:
        retire_permit(permit);raise


_CHILD_AUTHORITY=None


def consume_resolver_handoff(fd, manifest, source_sha, admission_path):
    global _CHILD_AUTHORITY
    require(_CHILD_AUTHORITY is None,'resolver authority already redeemed in this process')
    require(type(fd) is int and fd>=3,'one-use inherited live channel required')
    channel=None
    try:
        require(stat.S_ISSOCK(os.fstat(fd).st_mode),'live authority requires issued socketpair; pipe/file cannot authorize')
        channel=socket.socket(fileno=fd);channel.setblocking(False)
        frame=receive_frame(channel,time.monotonic(),lambda:None,prefilled=True)
        require(type(frame) is dict and set(frame)=={'claim_id','secret','source_sha256','admission_sha256','source_commit','deadline'},'malformed live capability')
        require(type(frame['secret']) is str and re.fullmatch('[0-9a-f]{64}',frame['secret']),'missing ephemeral issuer capability')
        secret=bytes.fromhex(frame['secret'])
        gate=runpy.run_path(str(HERE.parents[1]/'source_generation.py'));source=gate['validate_selection'](manifest,source_sha)
        require(frame['source_sha256']==source_sha,'handoff source mismatch')
        root=registered_root();path=root/'claims'/(ATTEMPT+'.claim.json');raw=protected_bytes(path);claim=gate['load'](raw)
        require(claim.get('claim_id')==frame['claim_id'] and claim.get('controller_pid')==os.getppid(),'handoff not from claimed parent')
        require(process_birth(os.getppid())==claim.get('controller_birth'),'wrong/stale issuer PID birth')
        require(claim.get('source_generation_sha256')==source_sha and claim.get('admission_sha256')==frame['admission_sha256'] and claim.get('source_commit')==frame['source_commit'],'claim binding mismatch')
        data,_,_=checked_go(admission_path,frame['admission_sha256'],source,frame['source_commit'],starting=False)
        require(type(frame['deadline']) in (int,float) and not isinstance(frame['deadline'],bool),'invalid exchange deadline')
        deadline=min(frame['deadline'],time.monotonic()+LIMITS['work_seconds'],time.monotonic()+(timestamp(data['valid_until_utc'])-utc()).total_seconds())
        def initial_guard():
            require(not os.path.lexists(path.parent/(ATTEMPT+'.retired.json')),'issuer retired')
            require(process_birth(os.getppid())==claim['controller_birth'],'issuer birth changed')
        initial_guard()
        require(not os.path.lexists(path.parent/(ATTEMPT+'.resolver-consumed.json')),'resolver redemption already consumed')
        request={'challenge':secrets.token_hex(32),'child_birth':process_birth(os.getpid())}
        grant={'request_sha256':digest(canonical(request)),'claim_sha256':digest(raw)}
        send_frame(channel,signed(secret,'request',request),deadline,initial_guard)
        require(authenticated(secret,'grant',receive_frame(channel,deadline,initial_guard))==grant,'wrong live grant')
        send_frame(channel,signed(secret,'ack',grant),deadline,initial_guard)
        require(authenticated(secret,'commit',receive_frame(channel,deadline,initial_guard))==grant,'wrong live commit')
        _CHILD_AUTHORITY={'channel':channel,'path':path,'raw':raw,'birth':request['child_birth'],'grant':grant,
            'claim':claim,'admission_path':admission_path,'admission_sha':frame['admission_sha256'],'source':source,'head':frame['source_commit']}
        guard_resolver_effects()
        return data
    except BaseException:
        if channel is not None:channel.close()
        else:os.close(fd)
        _CHILD_AUTHORITY=None
        raise


def guard_resolver_effects():
    """Before WORK effects, never owned cleanup: revoke first, then record."""
    context=_CHILD_AUTHORITY;require(context is not None,'no live child authority')
    try:peek=context['channel'].recv(1,socket.MSG_PEEK)
    except BlockingIOError:peek=None
    require(peek is None,'live issuer revoked/closed')
    path=context['path']
    with claim_lock(path):
        require(not os.path.lexists(path.parent/(ATTEMPT+'.retired.json')),'issuer retired before effects')
        require(protected_bytes(path)==context['raw'],'claim changed')
        marker=json.loads(protected_bytes(path.parent/(ATTEMPT+'.resolver-consumed.json')))
        require(marker['child_birth']==context['birth'] and marker['request_sha256']==context['grant']['request_sha256'] and marker['claim_sha256']==context['grant']['claim_sha256'],'redemption owner changed')
        require(process_birth(os.getpid())==context['birth'] and process_birth(os.getppid())==context['claim']['controller_birth'],'child/issuer birth changed')
        checked_go(context['admission_path'],context['admission_sha'],context['source'],context['head'],starting=False)
        # Recheck closure at the effect admission linearization point, under lock.
        try:peek=context['channel'].recv(1,socket.MSG_PEEK)
        except BlockingIOError:peek=None
        require(peek is None,'issuer revoked before effect admission')

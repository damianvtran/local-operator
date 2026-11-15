"""R20 real direct children, VERBATIM supervisor prefix; no engine/app effects.

Only fixture pwd-home/explicit fault points change. No getppid mock. Anonymous
socketpairs/pipes are local IPC, never listeners or Docker/network sockets.
"""
import copy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import unittest

import final_attempt_admission as auth
import test_final_attempt_admission as fixtures

HERE=Path(__file__).resolve().parent
PREFIX=(HERE/'RESOLVER-COMMAND.md').read_text().split('```python\n',1)[1].split('\n```',1)[0].split('pins = json.loads',1)[0]


class ActualChildAuthority(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.FinalAttemptTests();self.f.setUp();self.children=[];self.channels=[]

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
            child.communicate(timeout=5)
        for channel in self.channels:channel.close()
        self.f.tearDown()

    def permit(self):
        _,permit=self.f.claim();auth.begin_controller(permit);auth.begin_native_start(permit)
        return permit

    def spawn(self,fd,mode='normal',extra=''):
        code="""import os,sys,pwd,types
pwd.getpwuid=lambda uid:types.SimpleNamespace(pw_dir=sys.argv[1])
import final_attempt_admission as auth
"""+extra+'\n'
        if mode=='crash-ack':
            code+='''original=auth.send_frame
calls=0
def faulty(*args,**kwargs):
 global calls
 calls+=1
 if calls==2:raise RuntimeError('injected child exception before ACK')
 return original(*args,**kwargs)
auth.send_frame=faulty
'''
        code+='exec(compile('+repr(PREFIX)+",'VERBATIM-resolver-prefix','exec'))\n"
        if mode=='pause':code+="print('AUTH-READY',flush=True);sys.stdin.buffer.read(1)\n"
        code+="auth.guard_resolver_effects();print('INERT-EFFECT',flush=True)\n"
        if mode=='wrong-parent':
            code="import subprocess,sys,os;p=subprocess.run([sys.executable,'-B','-c',"+repr(code)+",sys.argv[1]],pass_fds=("+str(fd)+",),capture_output=True,text=True);os.close("+str(fd)+");sys.stdout.write(p.stdout);sys.stderr.write(p.stderr);sys.exit(p.returncode)"
        env=dict(os.environ,PYTHONPATH=str(HERE),BENCH_SOURCE_GENERATION=str(self.f.manifest),
            BENCH_SOURCE_GENERATION_SHA256=self.f.source_sha,BENCH_ADMISSION_PATH=str(self.f.path),
            BENCH_REGISTERED_ROOT=str(self.f.root),BENCH_FINAL_PERMIT_FD=str(fd))
        try:
            child=subprocess.Popen([sys.executable,'-B','-c',code,str(self.f.home)],cwd=HERE,env=env,
                pass_fds=(fd,),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                text=True,start_new_session=True)
        finally:os.close(fd)
        self.children.append(child);return child

    def done(self,child,accepted=False):
        out,err=child.communicate(timeout=12)
        self.assertEqual('INERT-EFFECT' in out,accepted,(child.returncode,err[-500:]))
        self.assertEqual(child.returncode==0,accepted)
        # Sanitized actual child receipts, never frames/secrets/GO/raw payloads.
        import hashlib
        receipt={'case':self._testMethodName,'exit':child.returncode,
                 'inert_effects':int('INERT-EFFECT' in out),'native_effects':[],
                 'verbatim_prefix_sha256':hashlib.sha256(PREFIX.encode()).hexdigest()}
        with (Path(os.environ['LOCAL_OPERATOR_SCRATCHPAD'])/'r20-child-green.jsonl').open('a') as log:
            log.write(json.dumps(receipt,sort_keys=True)+'\n')
        return out,err

    def frame(self,permit):
        fd=auth.issue_resolver_handoff(permit,deadline=time.monotonic()+10)
        self.channels.append(permit.channel)  # retain/reap the original endpoint when testing copies
        endpoint=socket.socket(fileno=fd);endpoint.setblocking(False)
        try:return auth.receive_frame(endpoint,time.monotonic(),lambda:None,prefilled=True)
        finally:endpoint.close()

    def cloned_channel(self,frame):
        parent,child=socket.socketpair();parent.setblocking(False);child.setblocking(False);self.channels.append(parent)
        auth.send_frame(parent,frame,time.monotonic()+2,lambda:None)
        return parent,child.detach()

    def legacy_pipe(self,claim):
        rd,wr=os.pipe()
        os.write(wr,json.dumps({'nonce':claim['claim_id'],'admission_sha256':self.f.go_sha,
            'source_sha256':self.f.source_sha,'source_commit':self.f.head}).encode());os.close(wr)
        return rd

    def test_actual_native_adapter_stages_prefix_and_services_live_issuer_with_monitors(self):
        from unittest.mock import patch
        import lifecycle_proposal as life
        permit=self.permit();ops=object.__new__(life.NativeOps)
        ops.root=self.f.root;ops.directory=self.f.root/'inert-controller';ops.directory.mkdir(mode=0o700)
        ops.a={'_source_generation':str(self.f.manifest),'_source_generation_sha256':self.f.source_sha}
        ops.admission_path=self.f.path;ops.final_permit=permit;ops.deadline=time.monotonic()+10
        ops.owner=None;ops.monitoring=False;ops.d=['inert-no-cli'];events=[]
        ops.now=time.monotonic;ops.check_disk=lambda:events.append('disk')
        ops.sample_swap=lambda phase:events.append('swap')
        ops.vm_helper=lambda owner:(events.append('rss-ownership') or {'rss_bytes':1})
        def command(argv,**kwargs):
            self.assertEqual(argv,['inert-no-cli','ps','-q']);events.append('foreign-fence');return (0,'','')
        ops.command=command
        staged="import pwd,types;pwd.getpwuid=lambda uid:types.SimpleNamespace(pw_dir="+repr(str(self.f.home))+")\n"+PREFIX+"\nguard_resolver_effects();print('INERT-EFFECT',flush=True)\n"
        original=Path.read_text
        def view(path,*args,**kwargs):
            return '```python\n'+staged+'\n```' if path==HERE/'RESOLVER-COMMAND.md' else original(path,*args,**kwargs)
        real_popen=subprocess.Popen
        def capture(*args,**kwargs):
            kwargs.update(stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            return real_popen(*args,**kwargs)  # actual child; only its receipt streams differ
        with patch.object(Path,'read_text',view),patch.object(life.subprocess,'Popen',side_effect=capture):
            ops.begin_resolver()
        self.children.append(ops.resolver);self.done(ops.resolver,True)
        self.assertEqual(events[0],'foreign-fence')
        for required in ('disk','swap','rss-ownership'):self.assertIn(required,events)
        self.assertTrue(auth.marker_path(permit,'resolver-consumed').exists())

    def test_original_actual_child_once_and_two_rebuilt_pipes_reject(self):
        permit=self.permit();fd=auth.issue_resolver_handoff(permit,deadline=time.monotonic()+10)
        child=self.spawn(fd);auth.serve_resolver_handoff(permit,child.pid);self.done(child,True)
        record=json.loads(auth.protected_bytes(auth.marker_path(permit,'resolver-consumed')))
        self.assertEqual(record['child_birth']['pid'],child.pid)
        auth.retire_permit(permit)
        for _ in range(2):self.done(self.spawn(self.legacy_pipe(permit.claim)),False)

    def test_postretire_old_real_frame_is_rejected(self):
        permit=self.permit();frame=self.frame(permit);auth.retire_permit(permit)
        _,fd=self.cloned_channel(frame);self.done(self.spawn(fd),False)
        self.assertNotIn('secret',json.dumps(permit.claim))
        self.assertNotIn(frame['secret'],permit.path.read_text())

    def test_fabricated_never_issued_claim_and_selfchosen_secret_no_live_registry(self):
        f=self.f;directory=f.root/'claims';directory.mkdir(mode=0o700)
        claim={'attempt_id':auth.ATTEMPT,'claim_id':'1'*64,'controller_pid':os.getpid(),
            'controller_birth':auth.process_birth(os.getpid()),'admission_sha256':f.go_sha,
            'source_generation_sha256':f.source_sha,'source_commit':f.head,'operator_decisions':auth.DECISIONS}
        auth.exclusive_record(directory/(auth.ATTEMPT+'.claim.json'),claim)
        frame={'claim_id':claim['claim_id'],'secret':os.urandom(32).hex(),'admission_sha256':f.go_sha,
            'source_sha256':f.source_sha,'source_commit':f.head,'deadline':time.monotonic()+3}
        parent,fd=self.cloned_channel(frame);child=self.spawn(fd)
        with self.assertRaises(ValueError):auth.serve_resolver_handoff(None,child.pid)
        parent.close();self.done(child,False);self.assertFalse(auth._ACTIVE)

    def test_revocation_after_authentication_before_effects(self):
        permit=self.permit();child=self.spawn(auth.issue_resolver_handoff(permit,deadline=time.monotonic()+10),'pause')
        auth.serve_resolver_handoff(permit,child.pid)
        import select
        self.assertTrue(select.select([child.stdout],[],[],5)[0]);self.assertEqual(child.stdout.readline().strip(),'AUTH-READY')
        auth.retire_permit(permit);child.stdin.write('x');child.stdin.flush();self.done(child,False)

    def test_concurrently_started_children_ephemeral_frame_copy_only_one_redemption(self):
        permit=self.permit();frame=self.frame(permit)
        first,fd1=self.cloned_channel(frame);second,fd2=self.cloned_channel(frame)
        a=self.spawn(fd1);b=self.spawn(fd2)  # both issue requests concurrently
        permit.channel=first;auth.serve_resolver_handoff(permit,a.pid);self.done(a,True)
        permit.channel=second
        with self.assertRaises(ValueError):auth.serve_resolver_handoff(permit,b.pid)
        self.done(b,False)
        self.assertEqual(json.loads(auth.protected_bytes(auth.marker_path(permit,'resolver-consumed')))['child_birth']['pid'],a.pid)

    def test_selfchosen_secret_cannot_authenticate_against_actual_live_registry(self):
        permit=self.permit();frame=self.frame(permit);frame['secret']=os.urandom(32).hex()
        parent,fd=self.cloned_channel(frame);permit.channel=parent;child=self.spawn(fd)
        with self.assertRaisesRegex(ValueError,'authentication failed'):
            auth.serve_resolver_handoff(permit,child.pid)
        self.done(child,False)
        self.assertFalse(auth.marker_path(permit,'resolver-consumed').exists())

    def test_wrong_child_birth_rejected_by_live_parent(self):
        permit=self.permit();fd=auth.issue_resolver_handoff(permit,deadline=time.monotonic()+10)
        extra="""original_birth=auth.process_birth
def altered(pid):
 value=original_birth(pid)
 if pid==os.getpid():value['start']='0.000000'
 return value
auth.process_birth=altered
"""
        child=self.spawn(fd,extra=extra)
        with self.assertRaises(ValueError):auth.serve_resolver_handoff(permit,child.pid)
        self.done(child,False)

    def test_stale_parent_birth_rejects(self):
        permit=self.permit();frame=self.frame(permit)
        altered=copy.deepcopy(permit.claim);altered['controller_birth']['start']='0.000000'
        permit.path.write_text(json.dumps(altered));permit.path.chmod(0o600)
        parent,fd=self.cloned_channel(frame);child=self.spawn(fd)
        with self.assertRaises(ValueError):auth.serve_resolver_handoff(permit,child.pid)
        parent.close();self.done(child,False)

    def test_changed_go_hash_and_wrong_actual_parent_reject(self):
        permit=self.permit();fd=auth.issue_resolver_handoff(permit,deadline=time.monotonic()+10)
        changed=copy.deepcopy(self.f.base);changed['desk_go']['receipt_id']='mutated-go-receipt'
        self.f.save(changed);child=self.spawn(fd)
        with self.assertRaises(ValueError):auth.serve_resolver_handoff(permit,child.pid)
        self.done(child,False)

    def test_wrong_actual_parent_rejects_verbatim_grandchild(self):
        permit=self.permit();child=self.spawn(auth.issue_resolver_handoff(permit,deadline=time.monotonic()+10),'wrong-parent')
        with self.assertRaises(ValueError):auth.serve_resolver_handoff(permit,child.pid)
        self.done(child,False)

    def test_consumed_live_issuer_cannot_authorize_fresh_socket_copy(self):
        permit=self.permit();frame=self.frame(permit)
        parent,fd=self.cloned_channel(frame);permit.channel=parent
        child=self.spawn(fd);auth.serve_resolver_handoff(permit,child.pid);self.done(child,True)
        _,fd=self.cloned_channel(frame);self.done(self.spawn(fd),False)
        self.assertIs(auth._ACTIVE[permit.nonce],permit)

    def test_expired_go_clock_rejects_actual_prefix(self):
        permit=self.permit();child=self.spawn(auth.issue_resolver_handoff(permit,deadline=time.monotonic()+10),
            extra="import datetime;auth.utc=lambda:datetime.datetime.now(datetime.UTC)+datetime.timedelta(hours=2)")
        with self.assertRaises(ValueError):auth.serve_resolver_handoff(permit,child.pid)
        self.done(child,False)

    def test_child_exception_consumes_redemption_and_retires_issuer(self):
        permit=self.permit();child=self.spawn(auth.issue_resolver_handoff(permit,deadline=time.monotonic()+10),'crash-ack')
        with self.assertRaises(ValueError):auth.serve_resolver_handoff(permit,child.pid)
        self.done(child,False)
        self.assertTrue(auth.marker_path(permit,'resolver-consumed').exists());self.assertTrue(auth.marker_path(permit,'retired').exists())
        with self.assertRaises(ValueError):auth.issue_resolver_handoff(permit)

    def test_partial_empty_regular_descriptors_and_exchange_deadline(self):
        for payload in (b'',(100).to_bytes(4,'big')+b'{}'):
            parent,child=socket.socketpair();self.channels.append(parent)
            if payload:parent.send(payload)
            self.done(self.spawn(child.detach()),False)
        fd=os.open(self.f.path,os.O_RDONLY);self.done(self.spawn(fd),False)
        permit=self.permit();child=self.spawn(auth.issue_resolver_handoff(permit,deadline=time.monotonic()+1.5))
        self.done(child,False)  # no server; existing work deadline may only be tightened

    def test_revocation_marker_failure_still_closes_live_authority(self):
        from unittest.mock import patch
        permit=self.permit();frame=self.frame(permit)
        with patch.object(auth,'exclusive_record',side_effect=OSError('injected fsync/journal failure')):
            with self.assertRaises(OSError):auth.retire_permit(permit)
        self.assertFalse(auth._ACTIVE);self.assertEqual(permit.secret,b'')
        with self.assertRaises(ValueError):auth.issue_resolver_handoff(permit)


if __name__=='__main__':unittest.main()
